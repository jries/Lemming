package org.lemming.utils;

import java.util.ArrayList;
import java.util.List;

import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.outputs.NullStoreWarning;

public class StoreSplitter<DataType> implements Runnable {

	DataType dtype;
	Store<DataType> store = new QueueStore<DataType>();
	List<Store<DataType>> splitter = new ArrayList<Store<DataType>>();
	
	@Override
	public void run() {
		
		if (store==null) {new NullStoreWarning(this.getClass().getName()); return;}
		
		while ((dtype = store.get()) != null){
			for (int size=splitter.size(), i=0; i<size; i++){
				splitter.get(i).put(dtype);
			}
		}
	}
	
	public void addOutput(Store<DataType> s) {
		splitter.add(s);
	}

	public void setInput(Store<DataType> s){
		store = s;
	}

}
