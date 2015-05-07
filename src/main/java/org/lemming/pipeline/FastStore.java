package org.lemming.pipeline;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import javolution.util.FastTable;

public class FastStore<DataType> implements Store<DataType> {
	
	private FastTable<DataType> q = new FastTable<DataType>();
	private final ReadWriteLock lock = new ReentrantReadWriteLock();
	
	/**
	 * 
	 */
	public FastStore(){
	}
	
	/**
	 * @return Returns the length of the queue.
	 */
	public int getLength() {
		return q.size();
	}

	@Override
	public void put(DataType el) {
		q.offer(el);
	}
	
	public DataType peek(){
		DataType res = null;
		lock.readLock().lock();
		try{
			while (!isEmpty()){
				res = q.peek();
				if (res != null) break;
			} 
		} finally {
			lock.readLock().unlock();
		}
		
		return res;
	}

	@Override
	public DataType get() {
		DataType res = null;
		lock.readLock().lock();
		try{
			while (!isEmpty()){
				res = q.poll();
				if (res != null) break;
			} 
		} finally {
			lock.readLock().unlock();
		}
		
		return res;
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}

}
