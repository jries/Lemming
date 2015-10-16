package org.lemming.interfaces;

import java.util.Map;


/**
 * Interface for all modules, inspired by the org.scijava.module interface
 * 
 * @author Ronny Sczech
 */

public interface ModuleInterface{
	
	public void cancel();
	
	public Object getInput(Integer key);
	
	public Map<Integer, Element> getInputs();
	
	public Object getOutput(Integer key);
	
	public Map<Integer, Element> getOutputs();
	
	public void setInput(Integer key, Store store);
	
	public void setInputs(Map<Integer, Store> storeMap);
	
	public void setOutput(Integer key, Store store);
	
	public void setOutputs(Map<Integer, Store> storeMap);
	
	public boolean check();
	
}
