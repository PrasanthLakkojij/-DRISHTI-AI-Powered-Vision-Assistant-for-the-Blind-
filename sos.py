

'''
from twilio.rest import Client

sid = "ACe4dc585d5e791912e0661c1b4477ac3c"
token = "abafdf8620f76808868e3af8600edb2e"

client = Client(sid, token)

message = client.messages.create(
    body="Hello! This message was sent from Python.",
    from_="whatsapp:+14155238886",  # Twilio WhatsApp sandbox number
    to="whatsapp:+919133773230"
)

print("Message sent:", message.sid)'''








from twilio.rest import Client

client = Client("ACe4dc585d5e791912e0661c1b4477ac3c", "abafdf8620f76808868e3af8600edb2e")

message = client.messages.create(
    body="Hello! This is a normal SMS from Python.",
    from_="+1 901 593 5722",
    to="+917842174988"
)

print(message.sid)








from twilio.rest import Client

client = Client("ACe4dc585d5e791912e0661c1b4477ac3c", "abafdf8620f76808868e3af8600edb2e")
num=["7842174988","9133773230"]
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