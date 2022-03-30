#if 0
#include <iostream>
using namespace std;

// 版本二
template<class T>
class SharpedPtrPro;

template<class T>
class Counter {
    friend class SharpedPtrPro<T>;

public:
    Counter(T* ptr = nullptr)
        : _ptr(ptr)
        , _pCount(1)
    {
        if (ptr)
        {
            _pCount = 1;
        }
        else {
            _pCount = 0;
        }
    }

    ~Counter() {
        delete _ptr;
        _ptr = nullptr;
    }

    // 获取引用计数
    int& UseCount()
    {
        return _pCount;
    }

    T* operator->()
    {
        return _ptr;
    }

    T& operator*()
    {
        return *_ptr;
    }

private:
    T* _ptr;
    int _pCount;
};

template<class T>
class SharpedPtrPro
{
public:
    SharpedPtrPro(T* ptr = nullptr) {
        _pCount = new Counter<T>(ptr);
    }

    SharpedPtrPro(const SharpedPtrPro<T>& ap) {
        _pCount = ap._pCount;
        ++_pCount->UseCount();
    }

    SharpedPtrPro& operator=(const SharpedPtrPro<T>& ap)
    {
        if (this != &ap)
        {
            if (_pCount)
            {
                Release();
            }

            ++ap._pCount->UseCount();
            _pCount = ap._pCount;
        }

        return *this;
    }

    // 析构函数
    ~SharpedPtrPro()
    {
        Release();
    }

    void Release()
    {
        if (0 == --_pCount->UseCount())
        {
            delete _pCount;
            _pCount = nullptr;
        }
    }

    // 为了使其更像一个指针，所完成的基本操作
    T* operator->()
    {
        return _pCount;
    }

    T& operator*()
    {
        return *_pCount;
    }

    int& UseCount()
    {
        return _pCount->UseCount();
    }

private:
    Counter<T>* _pCount;
};

// 版本一
template<class T>
class SharpedPtr
{
public:
    SharpedPtr(T* ptr = nullptr) // 构造函数
        : _ptr(ptr)
        , _pCount(nullptr)
    {
        if (ptr)
        {
            _pCount = new int(1);
        }
        else {
            _pCount = new int(0);
        }
    }

    // ① 拷贝构造是一次性的(也就是拷贝之前是没有对象的)，直接新的对象托管源对象所托管的对象
    SharpedPtr(const SharpedPtr<T>& ap) // 拷贝构造
        : _ptr(ap._ptr)
        , _pCount(ap._pCount)
    {
        if (_ptr)
        {
            ++UseCount();
        }
    }

    // ap1 = ap2; 赋值运算符重载
    // ① ap1与其他对象共享一块空间逻辑判断
    // ② 如果托管了其他对象, 需要脱离
    // ③ ap1指向新的托管对象，并将此托管对象的引用计数+1
    SharpedPtr& operator=(const SharpedPtr<T>& ap)
    {
        // ①
        if (_ptr == ap._ptr) {
            return *this;
        }

        // ②
        if (_ptr)
        {
            Release();
        }

        // ③
        _ptr = ap._ptr;
        _pCount = ap._pCount;
        ++UseCount();

        return *this;
    }

    // 析构函数
    ~SharpedPtr()
    {
        Release();
    }

    // 检查引用计数并释放空间
    void Release()
    {
        if (0 == --UseCount())
        {
            delete _pCount;
            _pCount = nullptr;
            delete _ptr;
            _ptr = nullptr;
        }
    }

    // 获取引用计数, 这种写法有点问题，外部可以修改UseCount
    int& UseCount()
    {
        return *_pCount;
    }

private:
    T*   _ptr;
    int* _pCount;
};

// 自己实现模板
template <class T>
class MySharedPtr {
public:
    MySharedPtr(T* ptr = nullptr) : m_ptr(ptr) {
        if (ptr == nullptr) {
            m_useCounts = new int(0);
        }
        else {
            m_useCounts = new int(1);
        }
    }

    MySharedPtr(const MySharedPtr<T>& ref) : 
        m_ptr(ref.m_ptr), m_useCounts(ref.m_useCounts) {
        if (m_ptr != nullptr) {
            (*m_useCounts)++;
        }
    }

    MySharedPtr& operator= (const MySharedPtr<T>& sPtr) {
        if (sPtr.m_ptr == m_ptr) {
            return *this;
        }

        if (m_ptr) {
            Release();
        }

        m_ptr = sPtr.m_ptr;
        m_useCounts = sPtr.m_useCounts;
        (*m_useCounts)++;

        return *this;
    }

    ~MySharedPtr() {
        Release();
    }

    int UseCount() {
        return *m_useCounts;
    }

private:
    void Release() {
        if (0 == -- (* m_useCounts)) {
            delete m_ptr;
            m_ptr = nullptr;

            delete m_useCounts;
            m_useCounts = nullptr;
        }
    }

private:
    int* m_useCounts { nullptr };
    T*   m_ptr       { nullptr };
};

class A {
public:
    A(int val) {
        m_value = val;
    }

private:
    int m_value = { 0 };
};

void main() {
    A* p = new A(10);
    A* p1 = new A(10);
   
    MySharedPtr <A> sp1(p), sp2(sp1), sp3(sp2);

    // 每个shared_ptr所指向的对象都有一个引用计数，它记录了有多少个shared_ptr指向自己; 注：sp.use_count()函数返回sp所指对象的引用计数
    // count作为堆内存所有智能指针共享同一个引用计数

    cout << sp1.UseCount() << endl; // 3
    cout << sp2.UseCount() << endl; // 3
    cout << sp3.UseCount() << endl; // 3
    
    MySharedPtr <A> sp4(p1);
    cout << sp4.UseCount() << endl; // 1
    
    sp4 = sp3;
    cout << sp4.UseCount() << endl; // 4
    cout << sp3.UseCount() << endl; // 4
}

#endif