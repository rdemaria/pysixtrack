import numpy as np

from .base_classes import Element
from . import elements
from .particles import Particles

from .loader_sixtrack import _expand_struct
from .loader_mad import iter_from_madx_sequence
from .closed_orbit import linearize_around_closed_orbit
from .closed_orbit import healy_symplectify
from .linear_normal_form import _linear_normal_form


# missing access to particles._m:
deg2rad = np.pi / 180.


class Line(Element):
    _description = [
        ("elements", "", "List of elements", ()),
        ("element_names", "", "List of element names", ()),
    ]
    _extra = []

    def __len__(self):
        assert len(self.elements) == len(self.element_names)
        return len(self.elements)

    def to_dict(self, keepextra=False):
        out = {}
        out["elements"] = [el.to_dict(keepextra) for el in self.elements]
        out["element_names"] = self.element_names[:]
        return out

    @classmethod
    def from_dict(cls, dct, keepextra=True):
        self = cls(elements=[], element_names=[])
        for el in dct["elements"]:
            eltype = getattr(elements, el["__class__"])
            newel = eltype.from_dict(el)
            self.elements.append(newel)
        self.element_names = dct["element_names"]
        return self

    def append_line(self, line):
        # Append the elements
        if type(line) is Line:
            # got a pysixtrack line
            self.elements += line.elements
        else:
            # got a different type of line (e.g. pybplep)
            for ee in line.elements:
                type_name = ee.__class__.__name__
                newele = getattr(elements, type_name)(**ee._asdict())
                self.elements.append(newele)

        # Append the names
        self.element_names += line.element_names

        assert len(self.elements) == len(self.element_names)

        return self

    def track(self, p):
        ret = None
        for el in self.elements:
            ret = el.track(p)
            if ret is not None:
                break
        return ret

    def track_elem_by_elem(self, p, start=True, end=False):
        out = []
        if start:
            out.append(p.copy())
        for el in self.elements:
            ret = el.track(p)
            if ret is not None:
                break
            out.append(p.copy())
        if end:
            out.append(p.copy())
        return out

    def insert_element(self, idx, element, name):
        self.elements.insert(idx, element)
        self.element_names.insert(idx, name)
        # assert len(self.elements) == len(self.element_names)
        return self

    def append_element(self, element, name):
        self.elements.append(element)
        self.element_names.append(name)
        # assert len(self.elements) == len(self.element_names)
        return self

    def get_length(self):
        thick_element_types = (elements.Drift, elements.DriftExact)

        ll = 0
        for ee in self.elements:
            if isinstance(ee, thick_element_types):
                ll += ee.length
        return ll

    def get_s_elements(self, mode="upstream"):
        thick_element_types = (elements.Drift, elements.DriftExact)

        assert mode in ["upstream", "downstream"]
        s_prev = 0
        s = []
        for ee in self.elements:
            if mode == "upstream":
                s.append(s_prev)
            if isinstance(ee, thick_element_types):
                s_prev += ee.length
            if mode == "downstream":
                s.append(s_prev)
        return s

    def remove_inactive_multipoles(self, inplace=False):
        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, (elements.Multipole)):
                aux = [ee.hxl, ee.hyl] + ee.knl + ee.ksl
                if np.sum(np.abs(np.array(aux))) == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.elements.clear()
            self.element_names.clear()
            self.append_line(newline)
            return self
        else:
            return newline

    def remove_zero_length_drifts(self, inplace=False):
        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, (elements.Drift, elements.DriftExact)):
                if ee.length == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.elements.clear()
            self.element_names.clear()
            self.append_line(newline)
            return self
        else:
            return newline

    def merge_consecutive_drifts(self, inplace=False):
        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if len(newline.elements) == 0:
                newline.append_element(ee, nn)
                continue

            if isinstance(ee, (elements.Drift, elements.DriftExact)):
                prev_ee = newline.elements[-1]
                prev_nn = newline.element_names[-1]
                if isinstance(prev_ee, (elements.Drift, elements.DriftExact)):
                    prev_ee.length += ee.length
                    prev_nn += nn
                else:
                    newline.append_element(ee, nn)
            else:
                newline.append_element(ee, nn)

        if inplace:
            self.elements.clear()
            self.element_names.clear()
            self.append_line(newline)
            return self
        else:
            return newline

    def get_elements_of_type(self, types):
        if not hasattr(types, "__iter__"):
            type_list = [types]
        else:
            type_list = types

        names = []
        elements = []
        for ee, nn in zip(self.elements, self.element_names):
            for tt in type_list:
                if isinstance(ee, tt):
                    names.append(nn)
                    elements.append(ee)

        return elements, names

    def get_element_ids_of_type(self, types, start_idx_offset=0):
        assert start_idx_offset >= 0
        if not hasattr(types, "__iter__"):
            type_list = [types]
        else:
            type_list = types
        elem_idx = []
        for idx, elem in enumerate(self.elements):
            for tt in type_list:
                if isinstance(elem, tt):
                    elem_idx.append(idx+start_idx_offset)
                    break
        return elem_idx

    def linear_normal_form(self, M):
        return _linear_normal_form(M)

    def find_closed_orbit_and_linear_OTM(
        self, p0c, guess=None, d=1.e-7, tol=1.e-10, max_iterations=20, longitudinal_coordinate='zeta'
    ):
        if guess is None:
            guess = [0., 0., 0., 0., 0., 0.]
        
        assert len(guess) == 6

        closed_orbit = np.array(guess).copy()
    
        canonical_conjugate_momentum = {'tau' : 'ptau', 'zeta' : 'delta', 'sigma' : 'psigma'}
    
        if longitudinal_coordinate not in ['tau', 'zeta', 'sigma']:
            raise Exception('Longitudinal variable not recognized in search of closed orbit')
    
        longitudinal_momentum = canonical_conjugate_momentum[longitudinal_coordinate]
    
        for i in range(max_iterations):
            new_closed_orbit, M = linearize_around_closed_orbit(
                self, closed_orbit, p0c, d, longitudinal_coordinate, longitudinal_momentum
            )

            error = np.linalg.norm( new_closed_orbit - closed_orbit )
    
            closed_orbit = new_closed_orbit
            if error < tol:
                print('Converged with approximate distance: {}'.format(error))
                _, M = linearize_around_closed_orbit(
                    self, closed_orbit, p0c, d, longitudinal_coordinate, longitudinal_momentum
                )
                return closed_orbit, healy_symplectify(M)
    
            print ('Closed orbit search iteration: {}'.format(i))
    
        print('WARNING!: Search did not converge, approximate distance: {}'.format(error))
        return closed_orbit, healy_symplectify(M)

    def find_closed_orbit(
            self, p0c, guess=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            method="Nelder-Mead", **kwargs
            ):
        def _one_turn_map(coord):
            pcl = Particles(p0c=p0c, **kwargs)
            pcl.x = coord[0]
            pcl.px = coord[1]
            pcl.y = coord[2]
            pcl.py = coord[3]
            pcl.zeta = coord[4]
            pcl.delta = coord[5]

            self.track(pcl)
            coord_out = np.array(
                [pcl.x, pcl.px, pcl.y, pcl.py, pcl.zeta, pcl.delta]
            )

            return coord_out

        def _CO_error(coord):
            return np.sum((_one_turn_map(coord) - coord) ** 2)

        if method == "get_guess":
            res = type("", (), {})()
            res.x = guess
        else:
            import scipy.optimize as so

            res = so.minimize(
                _CO_error, np.array(guess), tol=1e-20, method=method
            )

        pcl = Particles(p0c=p0c, **kwargs)

        pcl.x = res.x[0]
        pcl.px = res.x[1]
        pcl.y = res.x[2]
        pcl.py = res.x[3]
        pcl.zeta = res.x[4]
        pcl.delta = res.x[5]

        return pcl

    def enable_beambeam(self):

        for ee in self.elements:
            if isinstance(ee, (elements.BeamBeam4D, elements.BeamBeam6D)):
                ee.enabled = True

    def disable_beambeam(self):

        for ee in self.elements:
            if isinstance(ee, (elements.BeamBeam4D, elements.BeamBeam6D)):
                ee.enabled = False

    def beambeam_store_closed_orbit_and_dipolar_kicks(
        self,
        particle_on_CO,
        separation_given_wrt_closed_orbit_4D=True,
        separation_given_wrt_closed_orbit_6D=True,
    ):

        self.disable_beambeam()
        closed_orbit = self.track_elem_by_elem(particle_on_CO)

        self.enable_beambeam()

        for ie, ee in enumerate(self.elements):
            # to transfer to beambeam.py

            if ee.__class__.__name__ == "BeamBeam4D":
                if separation_given_wrt_closed_orbit_4D:
                    ee.x_bb += closed_orbit[ie].x
                    ee.y_bb += closed_orbit[ie].y

                # Evaluate dipolar kick
                ptemp = closed_orbit[ie].copy()
                ptempin = ptemp.copy()

                ee.track(ptemp)

                ee.d_px = ptemp.px - ptempin.px
                ee.d_py = ptemp.py - ptempin.py

            elif ee.__class__.__name__ == "BeamBeam6D":
                if not separation_given_wrt_closed_orbit_6D:
                    raise ValueError("Not implemented!")

                # Store closed orbit
                ee.x_co = closed_orbit[ie].x
                ee.px_co = closed_orbit[ie].px
                ee.y_co = closed_orbit[ie].y
                ee.py_co = closed_orbit[ie].py
                ee.zeta_co = closed_orbit[ie].zeta
                ee.delta_co = closed_orbit[ie].delta

                # Evaluate 6d kick on closed orbit
                ptemp = closed_orbit[ie].copy()
                ptempin = ptemp.copy()

                ee.track(ptemp)

                ee.d_x = ptemp.x - ptempin.x
                ee.d_px = ptemp.px - ptempin.px
                ee.d_y = ptemp.y - ptempin.y
                ee.d_py = ptemp.py - ptempin.py
                ee.d_zeta = ptemp.zeta - ptempin.zeta
                ee.d_delta = ptemp.delta - ptempin.delta

    @classmethod
    def from_sixinput(cls, sixinput, classes=elements):
        other_info = {}

        line_data, rest, iconv = _expand_struct(sixinput, convert=classes)

        ele_names = [dd[0] for dd in line_data]
        elements = [dd[2] for dd in line_data]

        line = cls(elements=elements, element_names=ele_names)

        other_info["rest"] = rest
        other_info["iconv"] = iconv

        line.other_info = other_info

        return line

    @classmethod
    def from_madx_sequence(
        cls,
        sequence,
        classes=elements,
        ignored_madtypes=[],
        exact_drift=False,
        drift_threshold=1e-6,
        install_apertures=False,
        apply_madx_errors=False,
    ):

        line = cls(elements=[], element_names=[])

        for el_name, el in iter_from_madx_sequence(
            sequence,
            classes=classes,
            ignored_madtypes=ignored_madtypes,
            exact_drift=exact_drift,
            drift_threshold=drift_threshold,
            install_apertures=install_apertures,
        ):
            line.append_element(el, el_name)

        if apply_madx_errors:
            line._apply_madx_errors(sequence)

        return line

    # error handling (alignment, multipole orders, ...):

    def find_element_ids(self, element_name):
        """Find element_name in this Line instance's
        self.elements_name list. Assumes the names are unique.

        Return index before and after the element, taking into account
        attached _aperture instances (LimitRect, LimitEllipse, ...)
        which would follow the element occurrence in the list.

        Raises IndexError if element_name not found in this Line.
        """
        # will raise error if element not present:
        idx_el = self.element_names.index(element_name)
        try:
            # if aperture marker is present
            idx_after_el = self.element_names.index(element_name + "_aperture") + 1
        except ValueError:
            # if aperture marker is not present
            idx_after_el = idx_el + 1
        return idx_el, idx_after_el

    def _add_offset_error_to(self, element_name, dx=0, dy=0):
        idx_el, idx_after_el = self.find_element_ids(element_name)
        xyshift = elements.XYShift(dx=dx, dy=dy)
        inv_xyshift = elements.XYShift(dx=-dx, dy=-dy)
        self.insert_element(idx_el, xyshift, element_name + "_offset_in")
        self.insert_element(
            idx_after_el + 1, inv_xyshift, element_name + "_offset_out"
        )

    def _add_aperture_offset_error_to(self, element_name, arex=0, arey=0):
        idx_el, idx_after_el = self.find_element_ids(element_name)
        idx_el_aper = idx_after_el - 1
        if not self.element_names[idx_el_aper] == element_name + "_aperture":
            # it is allowed to provide arex/arey without providing an aperture
            print('Info: Element', element_name, ': arex/y provided without aperture -> arex/y ignored')
            return
        xyshift = elements.XYShift(dx=arex, dy=arey)
        inv_xyshift = elements.XYShift(dx=-arex, dy=-arey)
        self.insert_element(idx_el_aper, xyshift, element_name + "_aperture_offset_in")
        self.insert_element(
            idx_after_el + 1, inv_xyshift, element_name + "_aperture_offset_out"
        )

    def _add_tilt_error_to(self, element_name, angle):
        '''Alignment error of transverse rotation around s-axis.
        The element corresponding to the given `element_name`
        gets wrapped by SRotation elements with rotation angle
        `angle`.

        In the case of a thin dipole component, the corresponding
        curvature terms in the Multipole (hxl and hyl) are rotated
        by `angle` as well.
        '''
        idx_el, idx_after_el = self.find_element_ids(element_name)
        element = self.elements[self.element_names.index(element_name)]
        if isinstance(element, elements.Multipole) and (
                element.hxl or element.hyl):
            dpsi = angle * deg2rad

            hxl0 = element.hxl
            hyl0 = element.hyl

            hxl1 = hxl0 * np.cos(dpsi) - hyl0 * np.sin(dpsi)
            hyl1 = hxl0 * np.sin(dpsi) + hyl0 * np.cos(dpsi)

            element.hxl = hxl1
            element.hyl = hyl1
        srot = elements.SRotation(angle=angle)
        inv_srot = elements.SRotation(angle=-angle)
        self.insert_element(idx_el, srot, element_name + "_tilt_in")
        self.insert_element(idx_after_el + 1, inv_srot, element_name + "_tilt_out")

    def _add_multipole_error_to(self, element_name, knl=[], ksl=[]):
        # will raise error if element not present:
        assert element_name in self.element_names
        element = self.elements[self.element_names.index(element_name)]
        # normal components
        knl = np.trim_zeros(knl, trim="b")
        if len(element.knl) < len(knl):
            element.knl += [0] * (len(knl) - len(element.knl))
        for i, component in enumerate(knl):
            element.knl[i] += component
        # skew components
        ksl = np.trim_zeros(ksl, trim="b")
        if len(element.ksl) < len(ksl):
            element.ksl += [0] * (len(ksl) - len(element.ksl))
        for i, component in enumerate(ksl):
            element.ksl[i] += component

    def _apply_madx_errors(self, madx_sequence):
        """Applies errors from MAD-X sequence to existing
        elements in this Line instance.

        Return names of MAD-X elements with existing align_errors
        or field_errors which were not found in the elements of
        this Line instance (and thus not treated).

        Example via cpymad:
            madx = cpymad.madx.Madx()

            # (...set up lattice and errors in cpymad...)

            seq = madx.sequence.some_lattice
            pysixtrack_line = pysixtrack.Line.from_madx_sequence(
                                    seq,
                                    apply_madx_errors=True
                              )
        """
        elements_not_found = []
        for element, element_name in zip(
                madx_sequence.expanded_elements,
                madx_sequence.expanded_element_names()
        ):
            if element_name not in self.element_names:
                if element.align_errors or element.field_errors:
                    elements_not_found.append(element_name)
                    continue

            if element.align_errors:
                # add offset
                dx = element.align_errors.dx
                dy = element.align_errors.dy
                if dx or dy:
                    self._add_offset_error_to(element_name, dx, dy)

                # add tilt
                dpsi = element.align_errors.dpsi
                if dpsi:
                    self._add_tilt_error_to(element_name, angle=dpsi / deg2rad)

                # add aperture-only offset
                arex = element.align_errors.arex
                arey = element.align_errors.arey
                if arex or arey:
                    self._add_aperture_offset_error_to(element_name, arex, arey)

                # check for errors which cannot be treated yet:
                for error_type in dir(element.align_errors):
                    if not error_type[0] == '_' and \
                            error_type not in ['dx', 'dy', 'dpsi', 'arex',
                                               'arey', 'count', 'index']:
                        print(
                            f'Warning: MAD-X error type "{error_type}"'
                            " not implemented yet."
                        )

            if element.field_errors:
                # add multipole error
                if any(element.field_errors.dkn) or \
                            any(element.field_errors.dks):
                    knl = element.field_errors.dkn
                    ksl = element.field_errors.dks
                    knl = knl[:np.amax(np.where(knl)) + 1]  # delete trailing zeros
                    ksl = ksl[:np.amax(np.where(ksl)) + 1]  # to keep order low
                    self._add_multipole_error_to(element_name, knl, ksl)

        return elements_not_found


elements.Line = Line
