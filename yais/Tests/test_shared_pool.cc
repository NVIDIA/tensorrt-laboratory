/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <chrono>

#include "glog/logging.h"

#include "YAIS/Pool.h"
#include "YAIS/ThreadPool.h"

using yais::Pool;
using yais::ThreadPool;

struct Object
{
    Object(std::string name) : m_Name(name) {}
    Object(Object&& other) : m_Name(std::move(other.m_Name)) {}
    ~Object() {
        LOG(INFO) << "Destroying Object " << m_Name;
    }

    std::string GetName() { return m_Name; }

  private:
    std::string m_Name;
};

int main(int argc, char* argv[])
{
    LOG(INFO) << "Shared Pool Test";

    auto pool = Pool<Object>::Create();
    auto workers = std::make_unique<ThreadPool>(1);

    // pool->Push(std::make_shared<Object>("Foo"));
    // EmplacePush will take either the ObjectType of the Pool
    // or it will take a list of arguments and pass those args to the
    // constructor of `new ObjectType(Args...)`
    pool->EmplacePush("Foo");
    pool->EmplacePush(new Object("Bar"));
    pool->Push(std::make_shared<Object>("Baz"));

    {
        auto obj = pool->Pop();
        LOG(INFO) << obj->GetName();
        workers->enqueue([obj]{
            LOG(INFO) << "in thread with obj " << obj->GetName();
            std::this_thread::sleep_for(std::chrono::seconds(2));
            LOG(INFO) << "still got obj " << obj->GetName();
        });
        LOG(INFO) << "end scope";
    }

    LOG(INFO) << "deleting pool";
    // Verifies that checked out objects maintain a shared_ptr to the pool,
    // thus ensuring that the pool does not get removed until the last object
    // has been returned
    pool.reset();
    LOG(INFO) << "waiting on threads";
    workers.reset();
    LOG(INFO) << "end program";
}
