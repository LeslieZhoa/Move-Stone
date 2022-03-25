import subprocess
import sys
import argparse
import os
import types
'''
check the path memory
'''
def arg_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='check the meomey of the path',
                        default= '/workspace')


    return parser.parse_args(args)


if __name__ == "__main__":

    args = arg_parser(sys.argv[1:])
    cmd = "getfattr -d -m '.*' %s"%(args.path)
    devNull = open(os.devnull, 'w')
    gitproc = subprocess.Popen(cmd,stdout = subprocess.PIPE,shell=True)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()
    git_diff = git_diff.decode().split('\n')
    print('memory: {}T'.format(float(git_diff[3].split('=')[-1][1:-1])/(1024**4)))
    