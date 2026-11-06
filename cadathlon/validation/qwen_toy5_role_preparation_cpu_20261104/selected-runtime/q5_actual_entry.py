"""Q5 future actual entry. Source preparation never admits backend or bodies."""
import argparse,sys
INFRASTRUCTURE_EXIT=70

def terminal_status(call):
    try:
        status=call()
        if type(status)is not int or status not in(0,1):raise RuntimeError('invalid normal terminal status')
        return status
    except BaseException as error:
        try:print(type(error).__name__,file=sys.stderr)
        except BaseException:pass
        return INFRASTRUCTURE_EXIT

def dispatch(args):
    # Deliberately before any helper, source/body read, numerical import or RNG.
    raise PermissionError('Q5 actual role closure, resource policy and independent phase grants remain unadmitted')

def main(argv=None):
    parser=argparse.ArgumentParser(description='Default-denied Q5 diagnostic actual role entry')
    parser.add_argument('--role',choices=('setup','server','client'),required=True)
    parser.add_argument('--source-root',required=True);parser.add_argument('--source-map',required=True)
    parser.add_argument('--source-map-sha256',required=True);parser.add_argument('--phase-packet',required=True)
    parser.add_argument('--phase-packet-sha256',required=True)
    args=parser.parse_args(argv);return terminal_status(lambda:dispatch(args))

if __name__=='__main__':raise SystemExit(main())
