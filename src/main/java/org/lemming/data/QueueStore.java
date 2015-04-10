package org.lemming.data;

import java.util.concurrent.LinkedBlockingQueue;

import org.lemming.interfaces.Peekable;
import org.lemming.interfaces.Store;

/**
 * The QueueStore implements a Store with a blocking queue. That is a Store that will block the reading thread if empty, until a new element is available.
 * A non-blocking version is the NonblockingQueueStore class.
 * 
 * @author Thomas Pengo
 *
 * @param <DataType> - data type
 */
public class QueueStore<DataType> implements Store<DataType>, Peekable<DataType> {

	private LinkedBlockingQueue<DataType> q = new LinkedBlockingQueue<DataType>();

	@Override
	public void put(DataType el) {
		q.add(el);
	}

	/**
	 * Note: blocks the caller if empty.
	 * 
	 */
	@Override
	public DataType get()  {
		try {
			return q.take();
		} catch (InterruptedException e) {
			return null;
		}
	}
	
	/**
	 * @return Returns the length of the queue.
	 */
	public int getLength() {
		return q.size();
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}
	
	@Override
	public Store<DataType> newPeek() {
		return new Store<DataType>() {

			//int curElement = 0;
			
			@Override
			public void put(DataType el) {
				q.add(el);
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
			
		};
	}
	
}
