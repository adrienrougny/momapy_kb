"""Configuration and fixtures for pytest."""

import os
import importlib.util
import urllib.parse

import pytest

HAS_TESTCONTAINERS = importlib.util.find_spec("testcontainers") is not None
HAS_FALKORDBLITE = importlib.util.find_spec("redislite") is not None


@pytest.fixture(scope="session")
def neo4j_container():
    """Start a Neo4j container for the test session."""
    if not HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed")
    import testcontainers.neo4j

    container = testcontainers.neo4j.Neo4jContainer("neo4j:5")
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="session")
def falkordb_container():
    """Start a FalkorDB container for the test session."""
    if not HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed")
    import testcontainers.core.container
    import testcontainers.core.waiting_utils

    container = testcontainers.core.container.DockerContainer(
        "falkordb/falkordb:latest"
    )
    container.with_exposed_ports(6379)
    container.start()
    testcontainers.core.waiting_utils.wait_for_logs(
        container, "Ready to accept connections"
    )
    yield container
    container.stop()


@pytest.fixture(scope="session")
def neo4j_backend(neo4j_container):
    """Create a Neo4j backend connected to the test container."""
    import momapy_kb.lpg.backends.neo4j

    if os.environ.get("CI"):
        return momapy_kb.lpg.backends.neo4j.Neo4jBackend(
            hostname="localhost",
            port=7687,
            username="neo4j",
            password="testpassword",
        )
    parsed = urllib.parse.urlparse(neo4j_container.get_connection_url())
    return momapy_kb.lpg.backends.neo4j.Neo4jBackend(
        hostname=parsed.hostname,
        port=parsed.port,
        protocol=parsed.scheme,
        username=neo4j_container.username,
        password=neo4j_container.password,
    )


@pytest.fixture(scope="session")
def falkordb_backend(falkordb_container):
    """Create a FalkorDB backend connected to the test container."""
    import momapy_kb.lpg.backends.falkordb

    if os.environ.get("CI"):
        return momapy_kb.lpg.backends.falkordb.FalkorDBBackend(
            hostname="localhost",
            port=6379,
            database="test",
        )
    return momapy_kb.lpg.backends.falkordb.FalkorDBBackend(
        hostname=falkordb_container.get_container_host_ip(),
        port=int(falkordb_container.get_exposed_port(6379)),
        database="test",
    )


@pytest.fixture(scope="session")
def falkordblite_backend():
    """Create a FalkorDBLite backend using a temp file."""
    if not HAS_FALKORDBLITE:
        pytest.skip("redislite not installed")
    import momapy_kb.lpg.backends.falkordblite

    backend = momapy_kb.lpg.backends.falkordblite.FalkorDBLiteBackend(
        path="/tmp/test_momapy_kb_falkordblite.db",
        database="test",
    )
    yield backend
    backend.close()


def _backend_params():
    """Build the list of available backend parameter names."""
    if not HAS_TESTCONTAINERS:
        return []
    params = ["neo4j", "falkordb"]
    if HAS_FALKORDBLITE:
        params.append("falkordblite")
    return params


@pytest.fixture(params=_backend_params())
def backend(request):
    """Parameterized fixture providing each available backend."""
    return request.getfixturevalue(f"{request.param}_backend")


@pytest.fixture(scope="session")
def neo4j_session(neo4j_backend):
    """Create a momapy_kb Session connected to Neo4j."""
    import momapy_kb.lpg.session

    session = momapy_kb.lpg.session.Session(neo4j_backend)
    session.__enter__()
    yield session
    session.__exit__(None, None, None)


@pytest.fixture(scope="session")
def falkordb_session(falkordb_backend):
    """Create a momapy_kb Session connected to FalkorDB."""
    import momapy_kb.lpg.session

    session = momapy_kb.lpg.session.Session(falkordb_backend)
    session.__enter__()
    yield session
    session.__exit__(None, None, None)


@pytest.fixture(scope="session")
def falkordblite_session(falkordblite_backend):
    """Create a momapy_kb Session connected to FalkorDBLite."""
    import momapy_kb.lpg.session

    session = momapy_kb.lpg.session.Session(falkordblite_backend)
    session.__enter__()
    yield session
    session.__exit__(None, None, None)


@pytest.fixture(params=_backend_params())
def session(request):
    """Parameterized fixture providing a session for each available backend."""
    return request.getfixturevalue(f"{request.param}_session")


@pytest.fixture
def clear_database(session):
    """Clear database before and after each test."""
    session.delete_all()
    yield
    session.delete_all()
