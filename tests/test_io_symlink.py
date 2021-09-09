from base import TestCase
from adaled.utils.io_ import _auto_symlink_path

class TestSymlinkCreation(TestCase):
    def test_auto_symlink_path_non_dir(self):
        def test(input, output, msg):
            self.assertEqual(_auto_symlink_path(input), output, msg)

        test('abc/def/somepath',
             'abc/def/somepath-latest',
             "Should add '-latest'.")
        test('abc/def/somepath.pt',
             'abc/def/somepath-latest.pt',
             "Should add '-latest' before the extension.")
        test('abc/def/somepath-{frame:05d}.pt',
             'abc/def/somepath-latest.pt',
             "Should remove {...}.")
        test('abc/def/somepath{frame:05d}.pt',
             'abc/def/somepath-latest.pt',
             "Should add a trailing dash if there is none.")

    def test_auto_symlink_path_dir(self):
        def test(input, output, msg):
            self.assertEqual(_auto_symlink_path(input), output, msg)

        test('abc/def/somepath/',
             'abc/def/somepath-latest',
             "Should add '-latest'.")
        test('abc/def/somepath-{frame:05d}/',
             'abc/def/somepath-latest',
             "Should remove {...}.")
        test('abc/def/somepath{frame:05d}/',
             'abc/def/somepath-latest',
             "Should add a trailing dash if there is none.")
        test('abc/def/somepath.some.thing/',
             'abc/def/somepath.some.thing-latest',
             "Dots have no special meaning for folders.")
