package org.lemming.modules;

import java.util.Map;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.pipeline.Store;

public class SliceSelector extends SingleRunModule {
	
	private String inputKey;
	private int counter=0;
	private boolean keep;
	private Store output;
	private int slice;
	
	public SliceSelector(int slice, boolean keep){
		this.slice = slice;
		this.keep = keep;
	}

	@Override
	protected void beforeRun(){ 
		inputKey = inputs.keySet().iterator().next();
		output = outputs.values().iterator().next();
		for ( String key : inputs.keySet()){
			while (inputs.get(key).isEmpty()) pause(10);
		}
	}
	
	@Override
	public void process(Map<String, Element> data) {
		Localization loc = (Localization) data.get(inputKey);
		if (loc==null) return;
		
		if (loc.getFrame()==slice){
			output.put(loc); // put ROI to output store
			counter++;
		}
		
		if (keep)
			inputs.get(inputKey).put(loc); //put it back to the input store
		
		if(loc.isLast())
			cancel();		
	}
	
	@Override
	protected void afterRun(){
	}
	
	public int getCounter()	{
		return counter;
	}

}
