package org.lemming.data;

import java.util.LinkedList;
import java.util.Queue;

import org.lemming.interfaces.Store;

/**
 * This is an implementation of a Store with a non-blocking queue. A non-blocking queue will return null if the queue is empty.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <DataType> - data type
 */
public class NonblockingQueueStore<DataType> implements Store<DataType> {

	private Queue<DataType> q = new LinkedList<>();
	
	@Override
	public void put(DataType el) {
		q.add(el);
	}

	/**
	 * Returns the last element in the queue, if there is one, or null if there is none.
	 */
	@Override
	public DataType get() {
		return q.poll();
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}

}
