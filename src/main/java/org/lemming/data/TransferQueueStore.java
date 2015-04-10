package org.lemming.data;

import java.util.concurrent.LinkedTransferQueue;

import org.lemming.interfaces.Store;

/**
 * The QueueStore implements a Store with a transfer queue. A producer will wait for the consumer to consume the elements in this queue.
 * 
 * @author Ronny Sczech
 *
 * @param <DataType> - data type
 */

public class TransferQueueStore<DataType> implements Store<DataType> {
	
	private final LinkedTransferQueue<DataType> q = new LinkedTransferQueue<DataType>();

	@Override
	public void put(DataType el) {
		q.put(el);
	}

	@Override
	public DataType get() {
		
		try {
			return q.take();		
		} catch (InterruptedException e) {
			e.printStackTrace();
			return null;
		}
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}
	
	/**
	 * @return Returns the length of the queue.
	 */
	public int getLength() {
		return q.size();
	}
}
