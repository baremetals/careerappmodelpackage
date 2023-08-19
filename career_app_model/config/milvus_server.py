from pymilvus import connections
import os


def connect_to_milvus(host='localhost', port='19530', user=None, password=None):
    # If not provided, try to get connection details from environment variables
    user = user or os.environ.get('MILVUS_USER', 'username')
    password = password or os.environ.get('MILVUS_PASSWORD', 'password')
    host = host or os.environ.get('MILVUS_HOST', 'localhost')
    port = port or os.environ.get('MILVUS_PORT', '19530')
    connections.connect(
        alias="default",
        user=user,
        password=password,
        host=host,
        port=port
    )
    print("Connected to Milvus successfully!")

