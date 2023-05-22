import asyncio
import pytest
from asyncio.streams import StreamReader, StreamWriter
from unittest import mock
from smartscan.TCPserver import TCPServer


@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_client_connection(event_loop):
    reader = mock.Mock(spec=StreamReader)
    writer = mock.Mock(spec=StreamWriter)

    async def mock_start_server():
        await asyncio.sleep(0.1)
        return reader, writer

    server = TCPServer('localhost', 12345)

    with mock.patch.object(asyncio, 'start_server', side_effect=mock_start_server):
        event_loop.run_until_complete(server.tcp_loop())

    yield reader, writer

    event_loop.run_until_complete(server.server.wait_closed())


@pytest.mark.asyncio
async def test_handle_client(mock_client_connection):
    reader, writer = mock_client_connection

    # Send a message
    message = 'Hello, server!'
    writer.read.side_effect = [message.encode('utf-8'), b'']

    await asyncio.sleep(0.1)  # Wait for the server to handle the message

    writer.write.assert_called_once_with(
        f'Received message "{message}" with checksum 5b985d33eafa947e3936629e0d35d01455da5f80e7f41e5df768592fde5e15dd\n'.encode('utf-8'))
    writer.drain.assert_called_once()


def test_run_server(event_loop):
    server = TCPServer('localhost', 12345)
    asyncio.ensure_future(server.tcp_loop())

    async def send_message():
        await asyncio.sleep(0.1)  # Wait for the server to start
        reader, writer = await asyncio.open_connection('localhost', 12345)
        writer.write(b'Hello, server!\n')
        await writer.drain()
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    event_loop.run_until_complete(send_message())


if __name__ == '__main__':
    pytest.main()
