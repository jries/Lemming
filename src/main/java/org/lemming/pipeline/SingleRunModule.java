package org.lemming.pipeline;

import org.lemming.interfaces.Element;

public abstract class SingleRunModule extends AbstractModule {

	@Override
	public void run() {

		if (!inputs.isEmpty() && !outputs.isEmpty()) {
			if (inputs.keySet().iterator().hasNext()){
				iterator = inputs.keySet().iterator().next();
				while (inputs.get(iterator).isEmpty())
					pause(10);
			}
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = nextInput();
				if (data != null) 
					newOutput(process(data));
			}
			
			afterRun();
			return;
		}
		if (!inputs.isEmpty()) {  // no outputs
			if (inputs.keySet().iterator().hasNext()){
				iterator = inputs.keySet().iterator().next();
				while (inputs.get(iterator).isEmpty())
					pause(10);
			}
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = nextInput();
				if (data != null) 
					process(data);
			}
			
			afterRun();
			return;
		}
		if (!outputs.isEmpty()) { // no inputs
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = process(null);
				newOutput(data);
			}
			afterRun();
			return;
		}
		service.shutdown();
	}

	protected void afterRun() {
	}

	protected void beforeRun() {
	}
	
}
