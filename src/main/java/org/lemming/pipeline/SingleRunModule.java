package org.lemming.pipeline;

import org.lemming.interfaces.Element;

/**
 * Single threaded modules
 * 
 * @author Ronny Sczech
 *
 */
public abstract class SingleRunModule extends AbstractModule {

	@Override
	public void run() {

		if (!inputs.isEmpty() && !outputs.isEmpty()) {
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			while (inputs.get(iterator).isEmpty())
				pause(10);
			
			beforeRun();
			while (running) {
				Element data = nextInput();
				if (data != null) 
					processData(data);
				else pause(10);
			}
			afterRun();
			return;
		}
		if (!inputs.isEmpty()) {  // no outputs
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			while (inputs.get(iterator).isEmpty())
				pause(10);
			
			beforeRun();
			while (running) {
				Element data = nextInput();
				if (data != null) 
					processData(data);
				else pause(10);
			}
			afterRun();
			return;
		}
		if (!outputs.isEmpty()) { // no inputs
			beforeRun();
			while (running) {
				Element data = processData(null);
				newOutput(data);
			}
			afterRun();
		}
	}

	protected void afterRun() {
	}

	protected void beforeRun() {
		start = System.currentTimeMillis();
		running = true;
	}
	
}
