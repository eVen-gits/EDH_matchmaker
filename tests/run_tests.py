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

def load_tests():
    # Discover tests in tests.test_core
    # We explicitly only want to run tests from test_core as requested
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_core')

    filtered_suite = unittest.TestSuite()

    # Filter out TestPerformance
    for test in suite:
        # test_core.py creates a suite where each item might be a TestSuite for a TestCase class
        # or individual tests. We need to handle nested suites.
        if isinstance(test, unittest.TestSuite):
            for t in test:
                # Check if the test belongs to TestPerformance class
                if t.__class__.__name__ == 'TestPerformance':
                    continue
                filtered_suite.addTest(t)
        else:
             if test.__class__.__name__ == 'TestPerformance':
                continue
             filtered_suite.addTest(test)

    return filtered_suite

if __name__ == '__main__':
    suite = load_tests()

    # Create test-reports directory if it doesn't exist
    if not os.path.exists('test-reports'):
        os.makedirs('test-reports')

    if xmlrunner:
        print("Running tests with XML generating runner...")
        runner = xmlrunner.XMLTestRunner(output='test-reports', outsuffix='')
    else:
        print("xmlrunner not found, running with standard TextTestRunner...")
        runner = unittest.TextTestRunner()

    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)
