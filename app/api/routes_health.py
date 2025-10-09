from fastapi import APIRouter

router = APIRouter()

@router.get('/health', summary="Health Check")
async def health_check():
    '''Endpoint to check the health of the application.'''
    return {"status": "ok"}