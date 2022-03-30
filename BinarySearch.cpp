#if 0
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// base
int binarySearch(vector<int> vec, int target) {
    if (vec.empty()) {
        return -1;
    }

    int start = 0;
    int end = vec.size() - 1;

    while (start + 1 < end) { // start < end 超时
        int mid = start + (end - start) / 2;
        cout << "start: " << start << " end: " << end << " mid: " << mid << endl;
        /*  在start < end && mid不加1条件下:
            start: 0 end: 9 mid: 4
            start: 4 end: 9 mid: 6
            start: 4 end: 6 mid: 5
            start: 4 end: 5 mid: 4
            start: 4 end: 5 mid: 4
            start: 4 end: 5 mid: 4
            start: 4 end: 5 mid: 4
            start: 4 end: 5 mid: 4
        */
        if (vec[mid] == target) {
            end = mid;   // 重复元素中的第一个下标， mid加1元素重复的情况会死循环
            /*
                end = mid + 1;   // 重复元素中的第一个下标， mid加1会死循环
                start: 0 end: 9 mid: 4
                start: 5 end: 9 mid: 7
                start: 5 end: 8 mid: 6
                start: 5 end: 7 mid: 6
                start: 5 end: 7 mid: 6
                start: 5 end: 7 mid: 6
                .
                .
                .
            */
        }
        else if (vec[mid] > target) {
            end = mid + 1;
            // or end = mid + 1
        }
        else {
            start = mid + 1;
            // or start = mid + 1
        }
    }

    if (target == vec[start]) {
        return start;
    }

    if (target == vec[end]) {
        return end;
    }

    return -1;
}

int main() {
    vector<int> v = { 0, 1, 2, 3, 4, 5 ,5 ,5, 5, 90 };
    cout << binarySearch(v, 5) << endl;
    return 0;
}
#endif