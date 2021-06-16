import sixtracktools
import pysixtrack

six = sixtracktools.SixInput(".")
line = pysixtrack.Line.from_sixinput(six)
iconv = line.other_info["iconv"]

sixdump = sixtracktools.SixDump101("res/dump.dat")[1::2]


def compare(prun, pbench):
    out = []
    for att in "x px y py delta sigma".split():
        vrun = getattr(prun, att)
        vbench = getattr(pbench, att)
        diff = vrun - vbench
        out.append(abs(diff))
        print(f"{att:<5} {vrun:22.13e} {vbench:22.13e} {diff:22.13g}")
    print(f"max {max(out):21.12e}")
    return max(out)


print("")
for ii in range(1, len(iconv)):
    jja = iconv[ii - 1]
    jjb = iconv[ii]
    prun = pysixtrack.Particles(**sixdump[ii - 1].get_minimal_beam())
    print(f"\n-----sixtrack={ii} sixtracklib={jja} --------------")
    # print(f"pysixtr {jja}, x={prun.x}, px={prun.px}")
    for jj in range(jja + 1, jjb + 1):
        label, elem = line.element_names[jj], line.elements[jj]
        elem.track(prun)
        print(f"{jj} {label},{str(elem)[:50]}")
    pbench = pysixtrack.Particles(**sixdump[ii].get_minimal_beam())
    # print(f"sixdump {ii}, x={pbench.x}, px={pbench.px}")
    print("-----------------------")
    out = compare(prun, pbench)
    print("-----------------------\n\n")
    if out > 1e-13:
        print("Too large discrepancy")
        break
