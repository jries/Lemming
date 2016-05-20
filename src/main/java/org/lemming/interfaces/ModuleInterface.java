package org.lemming.interfaces;

import java.util.Map;
import java.util.concurrent.ExecutorService;


/**
 * Interface for all modules, inspired by the org.scijava.module interface
 * 
 * @author Ronny Sczech
 */

public interface ModuleInterface extends Runnable{
	
	void cancel();
	
	Object getInput(Integer key);
	
	Map<Integer, Element> getInputs();
	
	Object getOutput(Integer key);
	
	Map<Integer, Element> getOutputs();
	
	void setInput(Integer key, Store store);
	
	void setInputs(Map<Integer, Store> storeMap);
	
	void setOutput(Integer key, Store store);
	
	void setOutputs(Map<Integer, Store> storeMap);
	
	void setOutput(Store s);

	void setInput(Store s);
	
	boolean check();

	void setService(ExecutorService service);
}
