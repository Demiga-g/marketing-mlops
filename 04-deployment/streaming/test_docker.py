import requests
# import lambda_function



event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49643887233201747061676272982557254274270719067415379970",
                "data": "ewogICAgICAgICJJRCI6IDQ5OTc5LAogICAgICAgICJZZWFyX0JpcnRoIjogMTkzNiwKICAgICAgICAiSW5jb21lIjogNTI0MjgxNiwKICAgICAgICAiS2lkaG9tZSI6IDQsICAgICAKICAgICAgICAiVGVlbmhvbWUiOiAyLAogICAgICAgICJSZWNlbmN5IjogMiwgCiAgICAgICAgIk1udFdpbmVzIjogODAwLAogICAgICAgICJNbnRGcnVpdHMiOiAxNDksCiAgICAgICAgIk1udE1lYXRQcm9kdWN0cyI6IDEwOTQsCiAgICAgICAgIk1udEZpc2hQcm9kdWN0cyI6IDMwMCwKICAgICAgICAiTW50U3dlZXRQcm9kdWN0cyI6IDE0OCwKICAgICAgICAiTW50R29sZFByb2RzIjogMTksCiAgICAgICAgIk51bURlYWxzUHVyY2hhc2VzIjogMiwKICAgICAgICAiTnVtV2ViUHVyY2hhc2VzIjogNSwKICAgICAgICAiTnVtQ2F0YWxvZ1B1cmNoYXNlcyI6IDE2LAogICAgICAgICJOdW1TdG9yZVB1cmNoYXNlcyI6IDIwLAogICAgICAgICJOdW1XZWJWaXNpdHNNb250aCI6IDQsCiAgICAgICAgIkNvbXBsYWluIjogMCwKICAgICAgICAiQWNjZXB0ZWRDbXAzIjogMCwKICAgICAgICAiQWNjZXB0ZWRDbXA0IjogMCwKICAgICAgICAiQWNjZXB0ZWRDbXA1IjogMCwKICAgICAgICAiQWNjZXB0ZWRDbXAxIjogMCwKICAgICAgICAiQWNjZXB0ZWRDbXAyIjogMCwKICAgICAgICAiWl9Db3N0Q29udGFjdCI6IDEsCiAgICAgICAgIlpfUmV2ZW51ZSI6IDEsCiAgICAgICAgIkVkdWNhdGlvbiI6ICIybiBDeWNsZSIsCiAgICAgICAgIk1hcml0YWxfU3RhdHVzIjogIk1hcnJpZWQiLAogICAgICAgICJEdF9DdXN0b21lciI6ICIyMDEyLTAzLTA1IgogICAgfQ==",
                "approximateArrivalTimestamp": 1692859080.319
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49643887233201747061676272982557254274270719067415379970",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::878104896345:role/lambda-kinesis-role",
            "awsRegion": "eu-north-1",
            "eventSourceARN": "arn:aws:kinesis:eu-north-1:878104896345:stream/response_events"
        }
    ]
}
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
response = requests.post(url, json=event)
print(response.json())

# result = lambda_function.lambda_handler(event, None)
# print(result)