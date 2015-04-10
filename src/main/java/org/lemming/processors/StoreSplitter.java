package org.lemming.processors;

import java.util.ArrayList;
import java.util.List;

import org.lemming.interfaces.Splitter;
import org.lemming.interfaces.Store;
import org.lemming.outputs.NullStoreWarning;

/**
 * @author Ronny Sczech
 *
 * @param <DataType> - data type
 */
public class StoreSplitter<DataType> implements Splitter<DataType> {

	DataType dtype;
	Store<DataType> store;
	List<Store<DataType>> splitter = new ArrayList<Store<DataType>>();
	
	@Override
	public void run() {
		
		if (store==null) {new NullStoreWarning(this.getClass().getName()); return;}
		        
		while (!store.isEmpty()){
			dtype = store.get();
			for (int size=splitter.size(), i=0; i<size; i++){
				splitter.get(i).put(dtype);
			}
		}
	
	}
	
	@Override
	public void addOutput(Store<DataType> s) {
		splitter.add(s);
	}
	
	@Override
	public void setInput(Store<DataType> s){
		store = s;
	}

}
