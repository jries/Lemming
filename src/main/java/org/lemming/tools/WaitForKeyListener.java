package org.lemming.tools;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class WaitForKeyListener implements KeyListener {
	
	private long delay = 1000;
	private ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor();
	private ScheduledFuture< ? > future;
	private Runnable command;


	public WaitForKeyListener(long delay, Runnable command) {
		this.delay = delay;
		this.command = command;
	}

	@Override
	public void keyTyped(KeyEvent event) {
		if (future != null && !future.isDone()) {
			future.cancel(false);
		}
		future = executor.schedule(command, delay, TimeUnit.MILLISECONDS);
	}

	@Override
	public void keyPressed(KeyEvent e) {
	}

	@Override
	public void keyReleased(KeyEvent e) {

	}

}
