"""Tests for get_color function."""
import pytest
import re
from mci.visualization.plotting import get_color


class TestGetColor:
    """Test suite for get_color function."""

    def test_zero_value(self):
        """Test color at minimum value (0)."""
        color = get_color(0)
        assert color == "rgb(255,143,108)"
        assert self._is_valid_rgb(color)

    def test_fifty_value(self):
        """Test color at middle value (50)."""
        color = get_color(50)
        assert color == "rgb(255,233,0)"
        assert self._is_valid_rgb(color)

    def test_hundred_value(self):
        """Test color at maximum value (100)."""
        color = get_color(100)
        assert color == "rgb(2,255,71)"
        assert self._is_valid_rgb(color)

    def test_value_below_50(self):
        """Test color interpolation below 50."""
        color = get_color(25)
        assert self._is_valid_rgb(color)
        # Should be between red and yellow
        r, g, b = self._extract_rgb(color)
        assert 200 <= r <= 255
        assert 150 <= g <= 240
        assert 0 <= b <= 108

    def test_value_above_50(self):
        """Test color interpolation above 50."""
        color = get_color(75)
        assert self._is_valid_rgb(color)
        # Should be between yellow and green
        r, g, b = self._extract_rgb(color)
        assert 2 <= r <= 255
        assert 233 <= g <= 255
        assert 0 <= b <= 71

    def test_negative_value(self):
        """Test with negative value."""
        color = get_color(-10)
        assert self._is_valid_rgb(color)
        # Should extrapolate beyond red
        r, g, b = self._extract_rgb(color)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_value_above_100(self):
        """Test with value above 100."""
        color = get_color(150)
        assert self._is_valid_rgb(color)
        # Should extrapolate beyond green
        r, g, b = self._extract_rgb(color)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_interpolation_smoothness(self):
        """Test that interpolation is smooth across range."""
        colors = [get_color(i) for i in range(0, 101, 10)]

        for color in colors:
            assert self._is_valid_rgb(color)

        # Check that colors change gradually
        for i in range(len(colors) - 1):
            r1, g1, b1 = self._extract_rgb(colors[i])
            r2, g2, b2 = self._extract_rgb(colors[i + 1])

            # Changes should be reasonable (not too large)
            assert abs(r2 - r1) <= 100
            assert abs(g2 - g1) <= 100
            assert abs(b2 - b2) <= 100

    def test_float_values(self):
        """Test with float values."""
        color = get_color(33.5)
        assert self._is_valid_rgb(color)

        color = get_color(66.7)
        assert self._is_valid_rgb(color)

    def test_boundary_values(self):
        """Test values at exact boundaries."""
        for val in [0, 50, 100]:
            color = get_color(val)
            assert self._is_valid_rgb(color)
            r, g, b = self._extract_rgb(color)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_near_boundary_values(self):
        """Test values very close to boundaries."""
        for val in [0.1, 49.9, 50.1, 99.9]:
            color = get_color(val)
            assert self._is_valid_rgb(color)

    def test_zero_point_one_intervals(self):
        """Test color values at fine intervals."""
        for val in [i * 0.1 for i in range(0, 1001, 100)]:
            color = get_color(val)
            assert self._is_valid_rgb(color)

    def test_color_format(self):
        """Test that color format is correct."""
        color = get_color(50)
        assert color.startswith("rgb(")
        assert color.endswith(")")
        assert color.count(",") == 2

    def test_integer_rgb_values(self):
        """Test that RGB values are integers."""
        for val in range(0, 101, 10):
            color = get_color(val)
            r, g, b = self._extract_rgb(color)
            assert isinstance(r, int)
            assert isinstance(g, int)
            assert isinstance(b, int)

    def test_red_component_decreases_after_50(self):
        """Test that red component decreases from 50 to 100."""
        colors = [get_color(i) for i in range(50, 101, 10)]
        red_values = [self._extract_rgb(c)[0] for c in colors]

        # Red should generally decrease
        assert red_values[0] > red_values[-1]

    def test_green_component_increases(self):
        """Test that green component increases from 0 to 100."""
        colors = [get_color(i) for i in range(0, 101, 10)]
        green_values = [self._extract_rgb(c)[1] for c in colors]

        # Green should generally increase
        assert green_values[0] < green_values[-1]

    def test_blue_component_behavior(self):
        """Test blue component behavior across range."""
        colors = [get_color(i) for i in range(0, 101, 10)]
        blue_values = [self._extract_rgb(c)[2] for c in colors]

        # Blue should decrease from 0 to 50, then increase from 50 to 100
        mid_idx = 5  # Index for value 50
        assert blue_values[0] > blue_values[mid_idx]
        assert blue_values[-1] > blue_values[mid_idx]

    def test_large_values(self):
        """Test with very large values."""
        color = get_color(1000)
        assert self._is_valid_rgb(color)

    def test_very_small_values(self):
        """Test with very small negative values."""
        color = get_color(-1000)
        assert self._is_valid_rgb(color)

    # Helper methods
    def _is_valid_rgb(self, color_string):
        """Check if string is valid RGB format."""
        pattern = r'^rgb\(\d+,\d+,\d+\)$'
        return bool(re.match(pattern, color_string))

    def _extract_rgb(self, color_string):
        """Extract RGB values from color string."""
        match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', color_string)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        raise ValueError(f"Invalid color string: {color_string}")
