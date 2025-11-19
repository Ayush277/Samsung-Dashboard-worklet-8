from api.simple import app

# Vercel expects this pattern
def handler(event, context):
    return app

# Alternative pattern Vercel might use
application = app