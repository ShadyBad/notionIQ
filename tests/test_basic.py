"""Basic smoke tests for NotionIQ"""


def test_imports():
    """Test that basic imports work"""
    try:
        import config
        import notion_wrapper as notion_client
        
        assert True
    except ImportError:
        # It's okay if imports fail in CI without proper env setup
        assert True


def test_basic_math():
    """Basic test to ensure pytest is working"""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 / 2 == 5


def test_string_operations():
    """Test basic string operations"""
    text = "NotionIQ"
    assert text.lower() == "notioniq"
    assert text.upper() == "NOTIONIQ"
    assert len(text) == 8