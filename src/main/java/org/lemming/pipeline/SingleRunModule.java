package org.lemming.pipeline;

import java.util.HashMap;
import java.util.Map;

public abstract class SingleRunModule extends AbstractModule {

	@Override
	public void run() {
		beforeRun();
		
		if (!inputs.isEmpty()){
			while (running) {
				if (Thread.currentThread().isInterrupted()) break;
				Map<String, Element> data = nextInput();
				process(data);
			}
			afterRun();	
			return;
		}
		if (!outputs.isEmpty()){
			while (running) {
				if (Thread.currentThread().isInterrupted()) break;
				Map<String, Element> data = new HashMap<>();
				process(data);
				
				for (String key : data.keySet())
					newOutput(key,data.get(key));
			}
			afterRun();	
			return;
		}			
	}

	public abstract void process(Map<String, Element> data);

	protected void afterRun() {
	}

	protected void beforeRun() {
	}

}
