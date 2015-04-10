package org.lemming.data;

import org.lemming.interfaces.Peekable;
import org.lemming.interfaces.Store;

import javolution.util.FastTable;

/**
 * @author Ronny Sczech
 *
 * @param <DataType> - data type
 */
public class FastStore<DataType> implements Store<DataType>, Peekable<DataType> {
	
	private FastTable<DataType> q = new FastTable<DataType>();
		
	/**
	 * 
	 */
	public FastStore(){
	}

	/**
	 * @return table
	 */
	public FastTable<DataType> getTable(){
		return q;
	}
	
	/**
	 * @param t - table to put
	 */
	public void putTable(FastTable<DataType> t){
		q.addAll(t);
	}
	
	@Override
	public void put(DataType el) {
		q.offer(el);	
	}

	@Override
	public boolean isEmpty() {
		return q.isEmpty();
	}

	@Override
	public DataType get() {
		DataType res = null;
		while (!isEmpty()){
			res = q.poll();
			if (res != null) break;
		} 
		return res;
	}
	
	/**
	 * @return Returns the length of the queue.
	 */
	public int getLength() {
		return q.size();
	}

	@Override
	public Store<DataType> newPeek() {
		return new Store<DataType>(){

			@Override
			public void put(DataType el) {
				q.offer(el);
			}

			@Override
			public DataType get() {
				DataType res = null;
				do {
				res = q.poll();
				} while (res  == null);
				return res;
			}

			@Override
			public boolean isEmpty() {
				return q.isEmpty();
			}
			
		};
	}

}
