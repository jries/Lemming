package org.lemming.pipeline;

import java.util.Map;

/**
 * Interface for all modules, inspired by the org.scijava.module interface
 * 
 * @author Ronny Sczech
 */

public interface ModuleInterface extends Runnable{
	
	/**
	 * @return Returns true if more outputs has to be generated.
	 */
	public boolean hasMoreOutputs();
	
	public void cancel();
	
	public Object getInput(String key);
	
	public Map<String, Object> getInputs();
	
	public Object getOutput(String key);
	
	public Map<String, Object> getOutputs();
	
	public void setInput(String key, Store<?> store);
	
	public void setInputs(Map<String, Store<?>> storeMap);
	
	public void setOutput(String key, Store<?> store);
	
	public void setOutputs(Map<String, Store<?>> storeMap);
	
}
