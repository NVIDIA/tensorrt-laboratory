/* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "glog/logging.h"
#include "trtlab/core/pool.h"
#include "gtest/gtest.h"

using namespace trtlab;

struct Object
{
    Object(std::string name) : m_Name(name), m_Original(name) {}
    ~Object()
    {
        DVLOG(2) << "Destroying Object " << m_Name;
    }

    Object(Object&& other) noexcept = default;
    Object& operator=(Object&& other) noexcept = default;

    void SetName(std::string name)
    {
        m_Name = name;
    }
    const std::string GetName() const
    {
        return m_Name;
    }

    void Reset()
    {
        m_Name = m_Original;
    }

private:
    std::string m_Name;
    std::string m_Original;
};

struct ShareableObject : public Object, public std::enable_shared_from_this<ShareableObject> 
{
    using Object::Object;
    auto Copy() { return shared_from_this(); }
};


class TestPool : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        p0 = Pool<Object>::Create();

        p1 = Pool<Object>::Create();
        p1->EmplacePush("Foo");

        p2 = Pool<Object>::Create();
        p2->EmplacePush("Foo");
        p2->EmplacePush("Bar");
    }

    virtual void TearDown() {}

    std::shared_ptr<Pool<Object>> p0;
    std::shared_ptr<Pool<Object>> p1;
    std::shared_ptr<Pool<Object>> p2;
};

TEST_F(TestPool, EmptyOnCreate)
{
    ASSERT_EQ(0, p0->Size());
}

TEST_F(TestPool, Push)
{
    ASSERT_EQ(0, p0->Size());

    p0->EmplacePush(Object("Baz"));
    ASSERT_EQ(1, p0->Size());
}

TEST_F(TestPool, Pop)
{
    ASSERT_EQ(1, p1->Size());

    auto obj = p1->Pop();
    ASSERT_TRUE(obj);
    ASSERT_EQ(0, p1->Size());
    obj.reset();
    ASSERT_EQ(1, p1->Size());

    {
        auto scoped_obj = p1->Pop();
        ASSERT_EQ(0, p1->Size());
    }
    ASSERT_EQ(1, p1->Size());
}

TEST_F(TestPool, PopOnReturn)
{
    auto foo = std::string("Foo");
    auto bar = std::string("Bar");
    {
        auto obj = p1->Pop([](Object& obj) { obj.Reset(); });
        ASSERT_TRUE(obj);
        ASSERT_EQ(foo, obj->GetName());
        obj->SetName(bar);
    }
    ASSERT_EQ(1, p1->Size());
    {
        auto obj = p1->Pop();
        ASSERT_TRUE(obj);
        ASSERT_EQ(foo, obj->GetName());
        obj->SetName(bar);
    }
    ASSERT_EQ(1, p1->Size());
    {
        auto obj = p1->Pop();
        ASSERT_TRUE(obj);
        ASSERT_EQ(bar, obj->GetName());
    }
    ASSERT_EQ(1, p1->Size());
}

TEST_F(TestPool, PopOnReturnWithCapture)
{
    auto foo = std::string("Foo");
    auto bar = std::string("Bar");

    auto obj = p1->Pop([](Object& obj) { obj.Reset(); });
    ASSERT_TRUE(obj);
    ASSERT_EQ(foo, obj->GetName());
    obj->SetName(bar);
    ASSERT_EQ(0, p1->Size());
    ASSERT_EQ(1, obj.use_count());

    // Capture obj in onReturn lambda
    auto from_p2_0 = p2->Pop([obj](Object& obj) {});
    ASSERT_EQ(2, obj.use_count());

    // Capture obj again a second onReturn lambda
    auto from_p2_1 = p2->Pop([obj](Object& obj) {});
    ASSERT_EQ(3, obj.use_count());

    // Free one of the resources that captured obj
    from_p2_0.reset();
    ASSERT_EQ(0, p1->Size());
    ASSERT_EQ(2, obj.use_count());
    ASSERT_EQ(bar, obj->GetName());

    // Free the original - it's still captured by from_p2_1
    obj.reset();
    ASSERT_EQ(0, p1->Size());

    // Free the last holder of obj
    from_p2_1.reset();
    ASSERT_EQ(1, p1->Size());

    {
        // Ensure the obj onReturn was called
        auto scoped = p1->Pop();
        ASSERT_EQ(foo, scoped->GetName());
    }
}

TEST_F(TestPool, PopWithoutReturn)
{
    ASSERT_EQ(1, p1->Size());

    auto obj = p1->PopWithoutReturn();
    ASSERT_TRUE(obj);
    ASSERT_EQ(0, p1->Size());
    obj.reset();
    ASSERT_EQ(0, p1->Size());
}

/*
TEST_F(TestPool, EnableSharedFromThis)
{
    auto p1 = Pool<std::shared_ptr<ShareableObject>>::Create();
    p1->Push(std::move(std::make_shared<ShareableObject>("Foo")));

    {
        auto scoped_obj = p1->Pop();
        ASSERT_EQ(scoped_obj.use_count(), 1);
        ASSERT_EQ(0, p1->Size());

        auto explicit_copy = scoped_obj;
        ASSERT_EQ(scoped_obj.use_count(), 2);
        ASSERT_EQ(explicit_copy.use_count(), 2);
        ASSERT_EQ(0, p1->Size());
        explicit_copy.reset();
        ASSERT_EQ(scoped_obj.use_count(), 1);
        ASSERT_EQ(0, p1->Size());

        auto copy_via_sft = scoped_obj->Copy();
        ASSERT_EQ(copy_via_sft.get(), scoped_obj.get());
        ASSERT_EQ(scoped_obj.use_count(), 2);
        ASSERT_EQ(copy_via_sft.use_count(), 2);

        scoped_obj.reset();
        ASSERT_EQ(copy_via_sft.use_count(), 1);
        ASSERT_EQ(0, p1->Size());
    }
}
*/