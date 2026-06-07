
 


from dotenv import load_dotenv
import os

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token  = os.getenv("TWILIO_AUTH_TOKEN")



num=["7842174988","9133773230"]





from twilio.rest import Client

client = Client(account_sid,auth_token)

message = client.messages.create(
    body="Hello! This is a normal SMS from Python.",
    from_="+1 901 593 5722",
    to=num
)

print(message.sid)








from twilio.rest import Client

client = Client(account_sid,auth_token)
call = client.calls.create(
    to=num,
    from_="+1 901 593 5722",
    twiml="""
<Response>
    <Say language="en-IN">
        his is an automated safety alert from the blind assistance system. The user may require assistance. Please contact them immediately..
    </Say>
</Response>
"""
)

print("Calling...")
