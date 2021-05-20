.PHONY: all test creds depend basic castor ffl configure test
  
all: test

depend:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Error - package python3 required, but not available."; exit 1; }

test: depend
	python3 -m pytest tests -srx -s
