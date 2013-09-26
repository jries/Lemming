package org.lemming.data;

public interface Store<DataType> {

	public void put(DataType el);
	
	public DataType get();
	
	boolean isEmpty();
}
