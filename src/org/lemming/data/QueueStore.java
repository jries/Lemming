package org.lemming.data;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class QueueStore<DataType> implements Store<DataType> {

	BlockingQueue<DataType> q = new LinkedBlockingQueue<DataType>();
	
	@Override
	public void put(DataType el) {
		q.add(el);
	}

	@Override
	public DataType get()  {
		try {
			return q.take();
		} catch (InterruptedException e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public int getLength() {
		return q.size();
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}

}
