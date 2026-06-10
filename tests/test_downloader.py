"""Tests for the downloader module."""

import pytest

from music_to_bass_score.downloader import validate_youtube_url


class TestValidateYoutubeUrl:
    def test_standard_watch_url(self):
        assert validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_short_url(self):
        assert validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")

    def test_shorts_url(self):
        assert validate_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")

    def test_invalid_url(self):
        assert not validate_youtube_url("https://example.com/video")

    def test_empty_string(self):
        assert not validate_youtube_url("")

    def test_url_without_https(self):
        assert validate_youtube_url("youtube.com/watch?v=dQw4w9WgXcQ")
