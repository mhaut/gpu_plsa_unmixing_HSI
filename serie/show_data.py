import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="Execution identifier")
parser.add_argument("image", help="Image to use")
parser.add_argument("id", help="Execution identifier")


args = parser.parse_args()

output_abundances_file_name = 'abundances_' + args.mode + '_' + args.image + '_' + args.id + '.npy'
output_endmembers_file_name = 'endmembers_' + args.mode + '_' + args.image + '_' + args.id + '.npy'


ab  = np.load(output_abundances_file_name)
end = np.load(output_endmembers_file_name)


print "endmember"
plt.plot(end)
plt.show()
print "abundances per endmember"
plt.imshow(np.hstack((ab[:,:,0],ab[:,:,1],ab[:,:,2])))
plt.show()
print "abundances"
plt.imshow(ab)
plt.show()
