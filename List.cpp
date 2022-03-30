#if 0
#include <iostream> 
using namespace std;

struct Node {
	int val;
	Node* next;
};

// reverse
Node* reverse(Node* head) {
	if (head == nullptr || head->next == nullptr) {
		return head;
	}

	Node* pre = nullptr;
	Node* cur = head;
	while (cur != nullptr) {
		Node* tmp = cur->next;
		cur->next = pre;
		pre = cur;
		cur = tmp;
	}

	return pre;
}

int main() {
	return 0;
}
#endif