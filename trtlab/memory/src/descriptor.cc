#include "descriptor.h"
#include "utils.h"

using namespace trtlab;
using namespace memory;

descriptor::descriptor() : m_storage(nullptr), m_size(0), m_alignment(0), m_data(nullptr) {}

descriptor::descriptor(std::shared_ptr<iallocator> alloc, std::size_t size, std::size_t alignment)
: m_storage(std::move(alloc)), m_size(size), m_alignment(alignment), m_data(m_storage->allocate(size, alignment))
{
}

void descriptor::release()
{
    if(m_storage && m_data)
    {
        m_storage->deallocate(m_data, m_size, m_alignment);
    }
}

DLContext descriptor::device_context() const
{
    DCHECK(m_storage);
    return m_storage->device_context();
}

std::shared_ptr<descriptor> descriptor::make_shared()
{
    return std::make_shared<descriptor>(std::move(*this));
}

std::ostream& trtlab::memory::operator<<(std::ostream& os, const descriptor& md)
{
    os << "[descriptor - addr: " << md.m_data << "; size: " << bytes_to_string(md.m_size) << "; alignment: " << md.m_alignment << "]";
    return os;
}