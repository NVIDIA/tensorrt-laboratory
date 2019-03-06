
import deploy_image_client

def main():
    router = deploy_image_client.ImageClient("localhost:50050")

    a = router.classify("model_a", "via_router_uuid1").get()
    b = router.classify("model_b", "via_router_uuid2").get()
    c = router.classify("model_c", "via_router_uuid3").get()

    assert a.uuid == "model_a"
    assert b.uuid == "model_b"
    assert c.uuid == "general_pool"

    a = router.detection("model_a", "via_router_uuid1").get()
    b = router.detection("model_b", "via_router_uuid2").get()
    c = router.detection("model_c", "via_router_uuid3").get()

    assert a.uuid == "general_pool"
    assert b.uuid == "model_b"
    assert c.uuid == "general_pool"

    print("\n**** Test Passed ****\n")

if __name__ == "__main__":
    main()
