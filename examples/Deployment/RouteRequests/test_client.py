import os

import deploy_image_client as cpp_client


def main():
    if not os.environ.get("TRTLAB_ROUTING_TEST"):
        raise RuntimeError(
            "Plese run this script in the environment setup by test_routing.sh")
    router = cpp_client.ImageClient("localhost:50050")

    print("Testing Classify RPC")

    a = router.classify("model_a", "via_router_uuid1").get()
    b = router.classify("model_b", "via_router_uuid2").get()
    c = router.classify("model_c", "via_router_uuid3").get()

    assert a.uuid == "model_a"
    assert b.uuid == "model_b"
    assert c.uuid == "general_pool"

    print("Testing Detection RPC")

    a = router.detection("model_a", "via_router_uuid1").get()
    b = router.detection("model_b", "via_router_uuid2").get()
    c = router.detection("model_c", "via_router_uuid3").get()

    assert a.uuid == "general_pool"
    assert b.uuid == "model_b"
    assert c.uuid == "general_pool"

    print("\n**** Test Passed ****\n")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("\n**** Error ****")
        print(e)
        print()
