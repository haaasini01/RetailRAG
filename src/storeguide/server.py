import uvicorn

from storeguide.config import HOST, PORT


def main() -> None:
    uvicorn.run("storeguide.app:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
