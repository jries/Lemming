package org.lemming.queue;

import java.util.concurrent.LinkedBlockingQueue;

/**
 * The QueueStore implements a Store with a blocking queue. That is a Store that will block the reading thread if empty, until a new element is available.
 * A non-blocking version is the NonblockingQueueStore class.
 * 
 * @author Thomas Pengo
 *
 * @param <DataType>
 */
public class QueueStore<DataType> implements Store<DataType>, Peekable<DataType> {

	LinkedBlockingQueue<DataType> q = new LinkedBlockingQueue<DataType>();
	
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
			e.printStackTrace();
			return null;
		}
	}
	
	/**
	 * Returns the length of the queue.
	 * 
	 * @return
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

			int curElement = 0;
			
			@Override
			public void put(DataType el) {
				q.add(el);
			}

			@Override
			public DataType get() {
				try {
					return q.take();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
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
