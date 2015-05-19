package org.lemming.pipeline;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import javolution.util.FastTable;

public class FastStore implements Store<Element> {
	
	private FastTable<Element> q = new FastTable<>();
	private final ReadWriteLock lock = new ReentrantReadWriteLock();
	
	/**
	 * 
	 */
	public FastStore(){
	}
	
	/**
	 * @return Returns the length of the queue.
	 */
	@Override
	public int getLength() {
		return q.size();
	}

	@Override
	public void put(Element el) {
		lock.writeLock().lock();
		try{
			q.offer(el);
		} finally {
			lock.writeLock().unlock();
		}
	}
	
	@Override
	public Element peek(){
		Element res = null;
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
	public Element get() {
		Element res = null;
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
