def replaceline_and_save(fname, findln, newline):
	if findln not in newline:
		raise ValueError('Detected inconsistency!!!!')
	
	with open(fname, 'r') as fid:
		lines = fid.readlines()
	
	found = False
	pos = None
	for ii, line in enumerate(lines):
		if findln in line:
			pos = ii
			found = True
			break
	
	if not found:
		raise ValueError('Not found!!!!')
		
	if '\n' in newline:
		lines[pos] = newline
	else:
		lines[pos] = newline+'\n'
	
	with open(fname, 'w') as fid:
		fid.writelines(lines)
