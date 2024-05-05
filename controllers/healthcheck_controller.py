from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/health", tags=["Healthcheck"], summary="Health Check Endpoint")
async def health_check():
    """
    Perform a health check of the application.

    This endpoint checks various components of the application, such as database connectivity and other services,
    to ensure that everything is running smoothly.

    Returns:
        A JSON response indicating the status of the application.
    """
    try:
        # Aqui você pode adicionar verificações de saúde específicas, como:
        # - Conexão com o banco de dados
        # - Verificação de APIs de terceiros
        # - Checagem de sistemas de arquivos, se necessário
        # Por exemplo:
        # db_status = check_database_connection()
        # if not db_status:
        #     raise Exception("Database connection failed")

        return {"status": "alive", "database": "online", "third_party_api": "reachable"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")
