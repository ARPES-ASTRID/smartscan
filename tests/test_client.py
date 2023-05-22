from typing import Tuple
import asyncio
import pytest
from unittest import mock
from smartscan.TCP import TCPClient
from asyncio.streams import StreamReader, StreamWriter


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    """
    Fixture to provide the asyncio event loop for the tests.
    """
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_server_connection(event_loop) -> Tuple[StreamReader, StreamWriter]:
    """
    Fixture to mock the server connection for testing the TCPClient.
    """
    reader = mock.Mock(spec=StreamReader)
    writer = mock.Mock(spec=StreamWriter)

    async def mock_connect() -> Tuple[StreamReader, StreamWriter]:
        await asyncio.sleep(0.1)
        return reader, writer

    client = TCPClient('localhost', 1234)

    with mock.patch.object(asyncio, 'open_connection', side_effect=mock_connect):
        event_loop.run_until_complete(client.connect())

    yield reader, writer

    event_loop.run_until_complete(writer.wait_closed())


@pytest.mark.asyncio
async def test_send_message(mock_server_connection) -> None:
    """
    Test case to verify the send_message method of TCPClient.
    """
    reader, writer = mock_server_connection

    # Send a message
    message = 'Hello, server!'
    writer.drain.return_value = asyncio.Future()
    writer.drain.return_value.set_result(None)
    reader.read.return_value = (f'Received message "{message}"\n').encode('utf-8')

    client = TCPClient('localhost', 1234)
    await client.send_message(message)

    writer.write.assert_called_once_with(message.encode('utf-8'))
    writer.drain.assert_called_once()


@pytest.mark.asyncio
async def test_receive_message(mock_server_connection) -> None:
    """
    Test case to verify the receive_message method of TCPClient.
    """
    reader, writer = mock_server_connection

    # Receive a message
    message = 'Hello, client!'
    reader.read.return_value = (f'{message}\n').encode('utf-8')

    client = TCPClient('localhost', 1234)
    received_message = await client.receive_message()

    reader.read.assert_called_once_with(1024)
    assert received_message == message


@pytest.mark.asyncio
async def test_connect_and_close(mock_server_connection) -> None:
    """
    Test case to verify the connect and close methods of TCPClient.
    """
    reader, writer = mock_server_connection

    client = TCPClient('localhost', 1234)
    await client.connect()

    asyncio.sleep.assert_called_once_with(0.1)
    asyncio.open_connection.assert_called_once_with('localhost', 1234)

    client.close()

    writer.close.assert_called_once()


if __name__ == '__main__':
    pytest.main()
