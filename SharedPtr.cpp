#if 0
#include <iostream>
using namespace std;

// �汾��
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

    // ��ȡ���ü���
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

    // ��������
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

    // Ϊ��ʹ�����һ��ָ�룬����ɵĻ�������
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

// �汾һ
template<class T>
class SharpedPtr
{
public:
    SharpedPtr(T* ptr = nullptr) // ���캯��
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

    // �� ����������һ���Ե�(Ҳ���ǿ���֮ǰ��û�ж����)��ֱ���µĶ����й�Դ�������йܵĶ���
    SharpedPtr(const SharpedPtr<T>& ap) // ��������
        : _ptr(ap._ptr)
        , _pCount(ap._pCount)
    {
        if (_ptr)
        {
            ++UseCount();
        }
    }

    // ap1 = ap2; ��ֵ���������
    // �� ap1������������һ��ռ��߼��ж�
    // �� ����й�����������, ��Ҫ����
    // �� ap1ָ���µ��йܶ��󣬲������йܶ�������ü���+1
    SharpedPtr& operator=(const SharpedPtr<T>& ap)
    {
        // ��
        if (_ptr == ap._ptr) {
            return *this;
        }

        // ��
        if (_ptr)
        {
            Release();
        }

        // ��
        _ptr = ap._ptr;
        _pCount = ap._pCount;
        ++UseCount();

        return *this;
    }

    // ��������
    ~SharpedPtr()
    {
        Release();
    }

    // ������ü������ͷſռ�
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

    // ��ȡ���ü���, ����д���е����⣬�ⲿ�����޸�UseCount
    int& UseCount()
    {
        return *_pCount;
    }

private:
    T*   _ptr;
    int* _pCount;
};

// �Լ�ʵ��ģ��
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

    // ÿ��shared_ptr��ָ��Ķ�����һ�����ü���������¼���ж��ٸ�shared_ptrָ���Լ�; ע��sp.use_count()��������sp��ָ��������ü���
    // count��Ϊ���ڴ���������ָ�빲��ͬһ�����ü���

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