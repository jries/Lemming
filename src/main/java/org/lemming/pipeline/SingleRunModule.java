package org.lemming.pipeline;

import org.lemming.interfaces.Element;

/**
 * Single threaded modules
 * 
 * @author Ronny Sczech
 *
 */
public abstract class SingleRunModule extends AbstractModule {
	
	private String name;
	private static int nr=0;
	
	public SingleRunModule(){
		name = this.getClass().getSimpleName();
	}

	@Override
	public void run() {
		Thread.currentThread().setName(name+nr++);
		if (!inputs.isEmpty() && !outputs.isEmpty()) {
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			while (inputs.get(iterator).isEmpty())
				pause(10);
			
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
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
				if (Thread.currentThread().isInterrupted())
					break;
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
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = processData(null);
				newOutput(data);
			}
			afterRun();
			return;
		}
		return;
	}

	protected void afterRun() {
	}

	protected void beforeRun() {
		start = System.currentTimeMillis();
	}
	
}
