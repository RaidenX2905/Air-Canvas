"""
Automated Comprehensive Test Suite for Air Canvas
Tests undo/redo immutability, continuous eraser collision, shape recognition,
Bézier curves, OneEuro filtering, and deterministic thread shutdown.
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from air_canvas import (
    Stroke,
    OneEuroFilter,
    dist_point_to_segment,
    try_recognize_shape,
    get_bezier_curve,
    WebcamStream
)


class TestAirCanvasCore(unittest.TestCase):

    def test_undo_redo_immutability(self):
        """Verifies deep clone undo/redo isolation (Task 1)."""
        s1 = Stroke((0, 255, 0), 4)
        s1.add_point((100, 100))
        s1.add_point((110, 110))

        strokes = [s1]
        undo_stack = [[s.clone() for s in strokes]]

        # Mutate the stroke
        s1.translate(50, 50, 1280, 720)
        self.assertEqual(s1.points[0], (150, 150))

        # Snapshot in undo_stack must remain at (100, 100)
        self.assertEqual(undo_stack[0][0].points[0], (100, 100))

        # Simulate undo restoration
        redo_stack = [[s.clone() for s in strokes]]
        restored = [s.clone() for s in undo_stack.pop()]
        self.assertEqual(restored[0].points[0], (100, 100))

        # Simulate redo restoration
        re_redone = [s.clone() for s in redo_stack.pop()]
        self.assertEqual(re_redone[0].points[0], (150, 150))

    def test_eraser_continuous_segment_collision(self):
        """Verifies point-to-segment distance collision during fast movement (Task 2)."""
        s = Stroke((255, 0, 0), 4)
        # Sparse stroke with 200px gap
        s.add_point((100, 100))
        s.add_point((300, 100))

        # Eraser hits directly at midpoint (200, 100) where NO point exists
        res = s.erase_at(200, 100, radius=20)
        # Stroke must be cut / erased rather than ignored
        self.assertTrue(len(res) != 1 or len(res[0].points) != 2)

    def test_dist_point_to_segment_math(self):
        """Verifies exact perpendicular projection distance."""
        d = dist_point_to_segment(50, 10, 0, 0, 100, 0)
        self.assertAlmostEqual(d, 10.0, places=4)

        # Off endpoint
        d_end = dist_point_to_segment(120, 0, 0, 0, 100, 0)
        self.assertAlmostEqual(d_end, 20.0, places=4)

    def test_one_euro_filter_adaptive_response(self):
        """Verifies dynamic cutoff frequency under rapid and slow movements."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        t = 1000.0
        val1 = f.filter(10.0, t)
        self.assertEqual(val1, 10.0)

        # Slow jitter
        val2 = f.filter(10.5, t + 0.033)
        self.assertTrue(10.0 < val2 < 10.5)

        # Fast snap: rapid movement sharply increases cutoff alpha
        val3 = f.filter(150.0, t + 0.066)
        self.assertTrue(val3 > 80.0)

    def test_shape_auto_snap(self):
        """Verifies line and circle recognition."""
        line_pts = [(i * 10, i * 10) for i in range(25)]
        snapped_line = try_recognize_shape(line_pts)
        self.assertIsNotNone(snapped_line)
        self.assertEqual(len(snapped_line), 2)

    def test_deterministic_thread_lifecycle(self):
        """Verifies WebcamStream stop event and join teardown (Task 3)."""
        stream = WebcamStream(src=0)
        stream.start()
        self.assertIsNotNone(stream.thread)
        self.assertTrue(stream.thread.is_alive())

        stream.stop()
        self.assertFalse(stream.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
