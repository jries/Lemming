package org.lemming.data;

import java.util.LinkedList;
import java.util.Queue;

public class NonblockingQueueStore<DataType> implements Store<DataType> {

	Queue<DataType> q = new LinkedList<>();
	
	@Override
	public void put(DataType el) {
		q.add(el);
	}

	@Override
	public DataType get() {
		return q.poll();
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}

}
