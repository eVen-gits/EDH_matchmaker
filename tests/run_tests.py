import sys
import os
import unittest

# Ensure root directory is in sys.path so we can import src and tests
sys.path.insert(0, os.getcwd())

# Try to import xmlrunner, fallback to standard runner if not found (for local testing without install)
try:
    import xmlrunner
except ImportError:
    xmlrunner = None

EXCLUDED_CLASSES = {"TestPerformance", "TestLarge"}


def load_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    filtered_suite = unittest.TestSuite()
    for test_group in suite:
        for test_case in test_group:
            if isinstance(test_case, unittest.TestSuite):
                for t in test_case:
                    if t.__class__.__name__ not in EXCLUDED_CLASSES:
                        filtered_suite.addTest(t)
            else:
                if test_case.__class__.__name__ not in EXCLUDED_CLASSES:
                    filtered_suite.addTest(test_case)

    return filtered_suite


if __name__ == "__main__":
    suite = load_tests()

    # Create test-reports directory if it doesn't exist
    if not os.path.exists("test-reports"):
        os.makedirs("test-reports")

    if xmlrunner:
        print("Running tests with XML generating runner...")
        runner = xmlrunner.XMLTestRunner(output="test-reports", outsuffix="", verbosity=2)
    else:
        print("xmlrunner not found, running with standard TextTestRunner...")
        runner = unittest.TextTestRunner(verbosity=2)

    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)
