try:
    f = open('testfile','r')
    f.write('This is the test file')
except IOError:
    print('Error')
else:
    print('Content')